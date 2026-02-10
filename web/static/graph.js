/**
 * RelationshipGraph - D3.js force-directed graph visualization for entity relationships
 */
class RelationshipGraph {
    constructor(container, data, detailPanelId) {
        this.container = container;
        this.data = data;
        this.detailPanelId = detailPanelId;
        
        // Graph dimensions
        this.width = container.clientWidth || 700;
        this.height = 450;
        
        // Node colors by type
        this.typeColors = {
            'Person': '#6366f1',      // Indigo
            'Organization': '#10b981', // Emerald
            'Location': '#f59e0b',     // Amber
            'default': '#8b5cf6'       // Violet
        };
        
        // Link colors by relationship type
        this.linkColors = {
            'Family': '#ec4899',       // Pink
            'Romantic': '#f43f5e',     // Rose
            'Social': '#22c55e',       // Green
            'Professional': '#3b82f6', // Blue
            'Case-related': '#f97316', // Orange
            'Transactional': '#eab308',// Yellow
            'default': '#6b7280'       // Gray
        };
        
        // State
        this.selectedNode = null;
        this.simulation = null;
        this.visibleLinkTypes = new Set();
        this.allLinkTypes = new Set();
        
        // Collect all link types
        this.data.links.forEach(link => {
            this.allLinkTypes.add(link.type || 'unknown');
            this.visibleLinkTypes.add(link.type || 'unknown');
        });
    }
    
    render() {
        // Clear container
        this.container.innerHTML = '';
        
        // Create SVG
        const svg = d3.select(this.container)
            .append('svg')
            .attr('width', '100%')
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height])
            .attr('class', 'relationship-graph-svg');
        
        // Add zoom behavior
        const g = svg.append('g').attr('class', 'graph-container');
        
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                g.attr('transform', event.transform);
            });
        
        svg.call(zoom);
        
        // Create arrow marker for directed edges
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#6b7280');
        
        // Create links
        const links = g.append('g')
            .attr('class', 'links')
            .selectAll('g')
            .data(this.data.links)
            .join('g')
            .attr('class', 'link-group');
        
        // Link lines
        const linkLines = links.append('line')
            .attr('class', 'link')
            .attr('stroke', d => this.getLinkColor(d.type))
            .attr('stroke-width', 2)
            .attr('stroke-opacity', 0.6);
        
        // Link labels (shown on hover)
        const linkLabels = links.append('text')
            .attr('class', 'link-label')
            .attr('text-anchor', 'middle')
            .attr('dy', -5)
            .text(d => d.type || 'connected')
            .style('font-size', '10px')
            .style('fill', '#a0a0a0')
            .style('opacity', 0)
            .style('pointer-events', 'none');
        
        // Show link label on hover
        links.on('mouseenter', function() {
            d3.select(this).select('.link-label').style('opacity', 1);
            d3.select(this).select('.link').attr('stroke-width', 3);
        }).on('mouseleave', function() {
            d3.select(this).select('.link-label').style('opacity', 0);
            d3.select(this).select('.link').attr('stroke-width', 2);
        });
        
        // Create nodes
        const nodes = g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(this.data.nodes)
            .join('g')
            .attr('class', 'node-group')
            .call(this.drag(this))
            .on('click', (event, d) => this.showNodeDetails(d));
        
        // Node circles
        nodes.append('circle')
            .attr('class', 'node')
            .attr('r', 20)
            .attr('fill', d => this.getNodeColor(d.type))
            .attr('stroke', '#1a1a1a')
            .attr('stroke-width', 2);
        
        // Node labels
        nodes.append('text')
            .attr('class', 'node-label')
            .attr('dy', 35)
            .attr('text-anchor', 'middle')
            .text(d => this.truncateName(d.name))
            .style('font-size', '11px')
            .style('fill', '#e8e8e8')
            .style('pointer-events', 'none');
        
        // Node initials (inside circle)
        nodes.append('text')
            .attr('class', 'node-initial')
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .text(d => this.getInitials(d.name))
            .style('font-size', '12px')
            .style('font-weight', '600')
            .style('fill', '#fff')
            .style('pointer-events', 'none');
        
        // Create force simulation
        this.simulation = d3.forceSimulation(this.data.nodes)
            .force('link', d3.forceLink(this.data.links)
                .id(d => d.id)
                .distance(120))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(40));
        
        // Update positions on tick
        this.simulation.on('tick', () => {
            linkLines
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            linkLabels
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);
            
            nodes.attr('transform', d => `translate(${d.x},${d.y})`);
        });
        
        // Store references for filtering
        this.links = links;
        this.linkLines = linkLines;
        this.nodes = nodes;
        this.svg = svg;
        this.g = g;
        
        // Add filter controls
        this.addFilterControls();
        
        // Initial center zoom
        svg.call(zoom.transform, d3.zoomIdentity.translate(0, 0).scale(0.9));
    }
    
    getNodeColor(type) {
        return this.typeColors[type] || this.typeColors.default;
    }
    
    getLinkColor(type) {
        return this.linkColors[type] || this.linkColors.default;
    }
    
    getInitials(name) {
        if (!name) return '?';
        const parts = name.split(' ').filter(p => p.length > 0);
        if (parts.length >= 2) {
            return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
        }
        return name.substring(0, 2).toUpperCase();
    }
    
    truncateName(name) {
        if (!name) return 'Unknown';
        if (name.length <= 15) return name;
        return name.substring(0, 12) + '...';
    }
    
    drag(graph) {
        function dragstarted(event) {
            if (!event.active) graph.simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) graph.simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
    
    showNodeDetails(node) {
        const detailPanel = document.getElementById(this.detailPanelId);
        if (!detailPanel) return;
        
        // Find relationships for this node
        const relationships = this.data.links.filter(
            link => link.source.id === node.id || link.target.id === node.id ||
                    link.source === node.id || link.target === node.id
        );
        
        let relationshipsHtml = '';
        if (relationships.length > 0) {
            relationshipsHtml = '<div class="detail-relationships"><strong>Relationships:</strong><ul>';
            relationships.forEach(rel => {
                const sourceName = typeof rel.source === 'object' ? rel.source.name : 
                    this.data.nodes.find(n => n.id === rel.source)?.name || rel.source;
                const targetName = typeof rel.target === 'object' ? rel.target.name : 
                    this.data.nodes.find(n => n.id === rel.target)?.name || rel.target;
                const otherPerson = sourceName === node.name ? targetName : sourceName;
                
                relationshipsHtml += `<li><span class="rel-type" style="color: ${this.getLinkColor(rel.type)}">${rel.type || 'connected'}</span> with <strong>${otherPerson}</strong>`;
                if (rel.description) {
                    relationshipsHtml += `<br><small>${rel.description}</small>`;
                }
                relationshipsHtml += '</li>';
            });
            relationshipsHtml += '</ul></div>';
        }
        
        detailPanel.innerHTML = `
            <div class="detail-header">
                <div class="detail-icon" style="background: ${this.getNodeColor(node.type)}">${this.getInitials(node.name)}</div>
                <div class="detail-title">
                    <h4>${node.name}</h4>
                    <span class="detail-type">${node.type || 'Person'}</span>
                </div>
                <button class="detail-close" onclick="this.parentElement.parentElement.innerHTML=''">&times;</button>
            </div>
            ${node.description ? `<p class="detail-description">${node.description}</p>` : ''}
            ${relationshipsHtml}
            ${node.sources && node.sources.length > 0 ? 
                `<div class="detail-sources"><small>Sources: ${node.sources.join(', ')}</small></div>` : ''}
        `;
        
        // Panel is visible when it has content (no class needed)
        
        // Highlight selected node
        this.nodes.selectAll('circle')
            .attr('stroke-width', d => d.id === node.id ? 4 : 2)
            .attr('stroke', d => d.id === node.id ? '#fff' : '#1a1a1a');
    }
    
    addFilterControls() {
        if (this.allLinkTypes.size <= 1) return; // Don't show filter if only one type
        
        const filterContainer = document.createElement('div');
        filterContainer.className = 'graph-filters';
        filterContainer.innerHTML = '<span class="filter-label">Filter by relationship:</span>';
        
        this.allLinkTypes.forEach(type => {
            const label = document.createElement('label');
            label.className = 'filter-checkbox';
            label.innerHTML = `
                <input type="checkbox" checked data-type="${type}">
                <span class="filter-color" style="background: ${this.getLinkColor(type)}"></span>
                <span class="filter-name">${type}</span>
            `;
            
            label.querySelector('input').addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.visibleLinkTypes.add(type);
                } else {
                    this.visibleLinkTypes.delete(type);
                }
                this.updateVisibility();
            });
            
            filterContainer.appendChild(label);
        });
        
        this.container.insertBefore(filterContainer, this.container.firstChild);
    }
    
    updateVisibility() {
        // Update link visibility
        this.links.style('display', d => 
            this.visibleLinkTypes.has(d.type || 'unknown') ? null : 'none'
        );
        
        // Find nodes that have visible connections
        const visibleNodeIds = new Set();
        this.data.links.forEach(link => {
            if (this.visibleLinkTypes.has(link.type || 'unknown')) {
                const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
                const targetId = typeof link.target === 'object' ? link.target.id : link.target;
                visibleNodeIds.add(sourceId);
                visibleNodeIds.add(targetId);
            }
        });
        
        // If all filters are on, show all nodes
        if (this.visibleLinkTypes.size === this.allLinkTypes.size) {
            this.nodes.style('opacity', 1);
        } else {
            // Dim nodes without visible connections
            this.nodes.style('opacity', d => 
                visibleNodeIds.has(d.id) ? 1 : 0.3
            );
        }
    }
}

// Export for use in app.js
window.RelationshipGraph = RelationshipGraph;
